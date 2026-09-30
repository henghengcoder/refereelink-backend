import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: root

    width: 1680
    height: 980
    minimumWidth: 1280
    minimumHeight: 820
    visible: true
    title: "Radar Dashboard"
    color: "#0b1016"

    onClosing: dashboard.shutdown()

    Rectangle {
        anchors.fill: parent
        color: "#0b1016"

        gradient: Gradient {
            GradientStop { position: 0.0; color: "#121a24" }
            GradientStop { position: 1.0; color: "#0b1016" }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 18

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 72
            radius: 22
            color: "#121923"
            border.width: 1
            border.color: "#26303d"

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 22
                anchors.rightMargin: 22
                spacing: 16

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 2

                    Label {
                        text: "Radar Dashboard"
                        color: "#f3f5f7"
                        font.pixelSize: 28
                        font.bold: true
                    }

                    Label {
                        text: "tracking dominant layout with live 2D projection"
                        color: "#9eabb8"
                        font.pixelSize: 13
                    }
                }

                Rectangle {
                    Layout.alignment: Qt.AlignVCenter
                    radius: 14
                    color: dashboard.finished ? "#1d3f32" : "#1a2430"
                    border.width: 1
                    border.color: dashboard.finished ? "#3d8f6a" : "#314253"
                    implicitWidth: statusLabel.implicitWidth + 28
                    implicitHeight: 36

                    Label {
                        id: statusLabel
                        anchors.centerIn: parent
                        text: dashboard.statusText
                        color: "#e8edf2"
                        font.pixelSize: 13
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 18

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.preferredWidth: 2.2
                radius: 26
                color: "#111821"
                border.width: 1
                border.color: "#26303d"

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 18
                    spacing: 12

                    Label {
                        text: "Tracking View"
                        color: "#f3f5f7"
                        font.pixelSize: 20
                        font.bold: true
                    }

                    Label {
                        text: "players and pitch keypoints"
                        color: "#90a0b0"
                        font.pixelSize: 12
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        radius: 20
                        color: "#0c1218"
                        border.width: 1
                        border.color: "#1e2935"
                        clip: true

                        Image {
                            anchors.fill: parent
                            anchors.margins: 10
                            source: "image://radarFrames/tracking?rev=" + dashboard.trackingRevision
                            fillMode: Image.PreserveAspectFit
                            cache: false
                            smooth: true
                            asynchronous: false
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.preferredWidth: 1.0
                radius: 26
                color: "#111821"
                border.width: 1
                border.color: "#26303d"

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 18
                    spacing: 12

                    Label {
                        text: "2D Projection"
                        color: "#f3f5f7"
                        font.pixelSize: 18
                        font.bold: true
                    }

                    Label {
                        text: "team positions on the mapped pitch"
                        color: "#90a0b0"
                        font.pixelSize: 12
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        radius: 20
                        color: "#0c1218"
                        border.width: 1
                        border.color: "#1e2935"
                        clip: true

                        Image {
                            anchors.fill: parent
                            anchors.margins: 10
                            source: "image://radarFrames/radar?rev=" + dashboard.radarRevision
                            fillMode: Image.PreserveAspectFit
                            cache: false
                            smooth: true
                            asynchronous: false
                        }
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 220
            radius: 24
            color: "#111821"
            border.width: 1
            border.color: "#26303d"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 10

                Label {
                    text: "Runtime Log"
                    color: "#f3f5f7"
                    font.pixelSize: 18
                    font.bold: true
                }

                Label {
                    text: "live model loading, projection status, and frame summaries"
                    color: "#90a0b0"
                    font.pixelSize: 12
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 18
                    color: "#0c1218"
                    border.width: 1
                    border.color: "#1e2935"

                    ScrollView {
                        anchors.fill: parent
                        anchors.margins: 10
                        clip: true
                        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                        TextArea {
                            text: dashboard.logText
                            readOnly: true
                            wrapMode: TextArea.NoWrap
                            color: "#dce2e8"
                            font.family: "monospace"
                            font.pixelSize: 13
                            selectByMouse: true
                            background: null
                        }
                    }
                }
            }
        }
    }
}
